from imap_processing.hit.l3.pha.pha_event_reader import RawPHAEvent, PHAWord, Detector


def create_raw_pha_event(particle_id=0, priority_buffer_num=0, stim_tag=False, haz_tag=False, time_tag=False,
                         a_b_side=False, has_unread=False, long_event_flag=False, culling_flag=False, spare=False,
                         pha_words=None) -> RawPHAEvent:
    if pha_words is None:
        pha_words = [
            PHAWord(adc_value=0, adc_overflow=False, detector=Detector.from_address(1), is_low_gain=False,
                    is_last_pha=False),
            PHAWord(adc_value=10, adc_overflow=False, detector=Detector.from_address(8), is_low_gain=True,
                    is_last_pha=True)
        ]

    return RawPHAEvent(particle_id=particle_id, priority_buffer_num=priority_buffer_num, haz_tag=haz_tag,
                       stim_tag=stim_tag,
                       time_tag=time_tag, long_event_flag=long_event_flag, a_b_side_flag=a_b_side,
                       has_unread_adcs=has_unread,
                       culling_flag=culling_flag, spare=spare, pha_words=pha_words)
